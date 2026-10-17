/* WebAssembly port, 2026-10-17. NetHack General Public License; see licenses/NetHack.txt. */
/* Replace the assembly context switch only; retain NLE's step/yield protocol. */
#include <stdlib.h>
#include <assert.h>
#include <emscripten/fiber.h>
#include "fcontext/fcontext.h"
#define ASYNC_STACK (256 * 1024)
typedef struct {
    emscripten_fiber_t fiber;
    fcontext_transfer_t transfer;
    pfn_fcontext entry;
    void *async_stack;
} context;
static context main_context;
static context *active;
static void enter(void *data) {
    context *ctx = data;
    ctx->entry(ctx->transfer);
    abort(); /* NLE must yield, never return from its game loop. */
}
fcontext_stack_t create_fcontext_stack(size_t size) {
    /* Some wasm frames are larger than their native counterparts. */
    size = size < 1024*1024 ? 1024*1024 : size;
    void *bottom = malloc(size);
    assert(bottom);
    return (fcontext_stack_t){(char *)bottom+size, size};
}
void destroy_fcontext_stack(fcontext_stack_t *stack) {
    free((char *)stack->sptr-stack->ssize);
}
fcontext_t make_fcontext(void *sp, size_t size, pfn_fcontext entry) {
    context *ctx = calloc(1, sizeof(context));
    ctx->entry = entry;
    ctx->async_stack = malloc(ASYNC_STACK);
    emscripten_fiber_init(&ctx->fiber, enter, ctx, (char *)sp-size, size,
                          ctx->async_stack, ASYNC_STACK);
    return ctx;
}
fcontext_transfer_t jump_fcontext(fcontext_t to, void *data) {
    if (!active) {
        main_context.async_stack = malloc(ASYNC_STACK);
        emscripten_fiber_init_from_current_context(&main_context.fiber,
                                                   main_context.async_stack, ASYNC_STACK);
        active = &main_context;
    }
    context *from = active, *target = to;
    target->transfer = (fcontext_transfer_t){from, data};
    active = target;
    emscripten_fiber_swap(&from->fiber, &target->fiber);
    active = from;
    return from->transfer;
}
