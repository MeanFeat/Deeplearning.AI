#pragma once

class Color {
public:
	Color(int inR, int inG, int inB, int inA) : r(inR), g(inG), b(inG), a(inA) { }
	int r, g, b, a;
};

void SDL_SetRenderDrawColor(SDL_Renderer *ren, Color c) {
	SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, c.a);
}