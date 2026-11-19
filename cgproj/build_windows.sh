#!/bin/bash
# Run this inside the "MSYS2 MINGW64" terminal (https://www.msys2.org).
# Produces a standalone car_studio.exe (no DLLs needed).
set -e
pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-glfw mingw-w64-x86_64-glew
g++ -std=c++17 -O2 -DGLEW_STATIC main.cpp -o car_studio.exe \
    -static -lglfw3 -lglew32 -lopengl32 -lgdi32
echo "Done: $(pwd)/car_studio.exe"
