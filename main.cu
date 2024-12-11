// /opt/cuda/bin/nvcc main.cu -o "GE" -diag-suppress 177 -diag-suppress 549 -ccbin=/usr/bin/clang -lSDL2 && time ./GE
#include <iostream>
#include <SDL2/SDL.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <thread>

#define width 1000
#define height 1000
#define G (double (6.6743e-11))

#define NUM_ASTEROIDS_X 1024 // Number of blocks
#define NUM_ASTEROIDS_Y 1024  // Block size
#define NUM_ASTEROIDS NUM_ASTEROIDS_X*NUM_ASTEROIDS_Y // Parameters for asteroid gen
#define MAX_XY 1e12
#define MAX_VELOCITY 1e4

SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
bool running = true;
bool paused = false;
long scale = 2e9;
int time_step = 1000;
long offset_x = 490;
long offset_y = 490;
double big_bodies[2*5] = {0, 0, 0, 0, 1.989e30, 7.78e11, 0, 0, 1.306e4, 1.898e27}; // Mass e-23, distance e-6.
double big_bodies_G_shite[2] = {};
double small_bodies[NUM_ASTEROIDS*4];
double * cuda_small_bodies;
double * cuda_big_bodies;
double * cuda_big_bodies_G_shite;

void draw_big_bodies(){
    SDL_Rect body_rect{0, 0, 10, 10};
    SDL_SetRenderDrawColor(renderer, 0,255,255,255);
    for (long i = 0; i < 2; i++){
        body_rect.x = int(long(big_bodies[i*5]) / scale + offset_x);
        body_rect.y = int(long(big_bodies[i*5+1]) / scale + offset_y);
        SDL_RenderFillRect(renderer, &body_rect);
    }
}
void draw_small_bodies(){
    SDL_SetRenderDrawColor(renderer, 255,255,255,255);
    for (long i = 0; i < NUM_ASTEROIDS; i++){
        SDL_RenderDrawPoint(renderer, int(long(small_bodies[i*4+0])/scale + offset_x), int(long(small_bodies[i*4+1])/scale + offset_y));
    }
}
void draw(){
    SDL_SetRenderDrawColor(renderer, 0,0,0,255);
    SDL_RenderClear(renderer);
    draw_big_bodies();
    draw_small_bodies();
    SDL_RenderPresent(renderer);
};

double mag(double x, double y){
    return sqrt((x*x)+(y*y));
}

void big_phys(){
    for (long i = 0; i < 2; i++)for (long j = 0; j < 2; j++){
            if (i == j) { continue;}
            double r = mag(big_bodies[i*5+0]-big_bodies[j*5+0], big_bodies[i*5+1]-big_bodies[j*5+1]);
            double acc_mag_over_r = (-G * big_bodies[j*5+4] ) / (r*r*r);
            big_bodies[i*5+2]+= acc_mag_over_r*(big_bodies[i*5+0]-big_bodies[j*5+0])*time_step;
            big_bodies[i*5+3]+= acc_mag_over_r*(big_bodies[i*5+1]-big_bodies[j*5+1])*time_step;
    };

    for (long i = 0; i < 2; i++){
        big_bodies[i*5+0]+= big_bodies[i*5+2]*time_step;
        big_bodies[i*5+1]+= big_bodies[i*5+3]*time_step;
    }
}
void randomise_asteroids() {
    std::srand(static_cast<unsigned>(std::time(0)));
    for (long i = 0; i < NUM_ASTEROIDS; ++i) {
        // Randomize position
        double angle = (std::rand() / (double)RAND_MAX) * 2 * M_PI;
        double distance = ((std::rand() / (double)RAND_MAX)*0.5 + 0.5) * MAX_XY; // Ensure non-zero distance
        small_bodies[i * 4 + 0] = distance * std::cos(angle); // x
        small_bodies[i * 4 + 1] = distance * std::sin(angle); // y

        // Compute velocity for circular orbit
        double speed = MAX_VELOCITY * (distance / MAX_XY);
        small_bodies[i * 4 + 2] = -speed * std::sin(angle); // vx (perpendicular)
        small_bodies[i * 4 + 3] = speed * std::cos(angle);  // vy (perpendicular)
    }
}

__device__ double mag_dev(double x, double y){
    return sqrt((x*x)+(y*y));
}

__global__ void asteriod_phys(double * small_bodies, double * big_bodies, double * big_bodies_G_shite, int time_step){
    long idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx >= NUM_ASTEROIDS)return;

    for (long i = 0; i < 2; i++){
        double r_x = small_bodies[idx*4+0]-big_bodies[i*5+0];
        double r_y = small_bodies[idx*4+1]-big_bodies[i*5+1];
        double r = mag_dev(r_x, r_y);
        if (r < (i == 0)? 5e8 : 1e7){

        }

        double acc_mag_over_r = big_bodies_G_shite[i] / (r*r*r); // -G * big_bodies[n] is recomputed for every asteroid. TODO: Fix that.
        small_bodies[idx*4+2]+= acc_mag_over_r*(r_x)*time_step;
        small_bodies[idx*4+3]+= acc_mag_over_r*(r_y)*time_step;
    };
    small_bodies[idx*4+0] += small_bodies[idx*4+2]*time_step;
    small_bodies[idx*4+1] += small_bodies[idx*4+3]*time_step;
}

void small_phys(){
    if (paused) return;
    cudaMalloc(&cuda_big_bodies, sizeof (big_bodies));
    cudaMemcpy(cuda_big_bodies, &big_bodies, sizeof (big_bodies), cudaMemcpyHostToDevice);
    for (int i = 0; i < 2; i++){
        big_bodies_G_shite[i] = (-G * big_bodies[i*5+4] );
    }
    cudaMalloc(&cuda_big_bodies_G_shite, sizeof (big_bodies_G_shite));
    cudaMemcpy(cuda_big_bodies_G_shite, &big_bodies_G_shite, sizeof (big_bodies_G_shite), cudaMemcpyHostToDevice);

    asteriod_phys<<<NUM_ASTEROIDS_X, NUM_ASTEROIDS_Y>>>(cuda_small_bodies, cuda_big_bodies, big_bodies_G_shite, time_step);
    cudaDeviceSynchronize();
}

void fixed_update(){
    while (running) {
        big_phys();
        small_phys();
    }
}

int main() {
    randomise_asteroids();

    cudaMalloc(&cuda_small_bodies, sizeof (small_bodies));
    cudaMemcpy(cuda_small_bodies, &small_bodies, sizeof (small_bodies), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_big_bodies, sizeof (big_bodies));
    cudaMemcpy(cuda_big_bodies, &big_bodies, sizeof (big_bodies), cudaMemcpyHostToDevice);

//    srand((unsigned) time(NULL));
    std::cout << "Asteroid sim... (Can't spell it :3)" << std::endl;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);
    SDL_RenderSetScale(renderer,1,1);
    SDL_SetWindowTitle(window, "Asteroid Sim");


    std::thread physicsThread(&fixed_update);
    int a,b=0;


    while (running) {
        a = SDL_GetTicks();
        if (a-b > 1000/60){
            cudaMemcpy(small_bodies, cuda_small_bodies, sizeof(small_bodies), cudaMemcpyDeviceToHost);
            draw();
            b=a;
        }
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym ==SDLK_SPACE)paused = !paused;
                    if (event.key.keysym.sym ==SDLK_DOWN){time_step*=0.1;};
                    if (event.key.keysym.sym ==SDLK_UP){time_step*=10;};
                    if (event.key.keysym.sym ==SDLK_LEFT){scale*=0.5;};
                    if (event.key.keysym.sym ==SDLK_RIGHT){scale*=2;};
                    if (event.key.keysym.sym ==SDLK_w){offset_y-=100;}
                    if (event.key.keysym.sym ==SDLK_s){offset_y+=100;}
                    if (event.key.keysym.sym ==SDLK_a){offset_x-=100;}
                    if (event.key.keysym.sym ==SDLK_d){offset_x+=100;}
                    break;
            }
        }
    }
    physicsThread.join();

    return 0;
}
