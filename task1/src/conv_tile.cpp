// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    // conv_naive(in, out, ker, H, W, K);
    const int p = K/2;
    const int stride = W + 2*p;
    const int tile_H = 16, tile_W = 64;
    for(int y=0; y<H; y++) {
        for(int x=0; x<W; x++) out[(y*W + x)] = 0.0f;
    }
    for(int y=0; y<H; y+=tile_H) {
        for(int x=0; x<W; x+=tile_W) {
            const int y_max = (y + tile_H < H) ? y + tile_H : H;
            const int x_max = (x + tile_W < W) ? x + tile_W : W;
            for(int ty=0; ty<K; ty++) {
                for(int tx=0; tx<K; tx++) {
                    const float tval = ker[(ty*K + tx)];
                    for(int oy=y; oy<y_max; oy++) {
                        for(int ox=x; ox<x_max; ox++) {
                            out[(oy*W + ox)] += in[((oy + ty)*stride + (ox + tx))]*tval;
                        }
                    }
                }
            }
        }
    }
}
