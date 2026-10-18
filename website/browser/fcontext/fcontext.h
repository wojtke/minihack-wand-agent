/* WebAssembly port, 2026-10-17. NetHack General Public License; see licenses/NetHack.txt. */
/* fcontext-compatible interface backed by Emscripten fibers. */
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef void *fcontext_t;
typedef struct { fcontext_t ctx; void *data; } fcontext_transfer_t;
typedef struct { void *sptr; size_t ssize; } fcontext_stack_t;
typedef void (*pfn_fcontext)(fcontext_transfer_t);
fcontext_transfer_t jump_fcontext(fcontext_t to, void *data);
fcontext_t make_fcontext(void *sp, size_t size, pfn_fcontext entry);
fcontext_stack_t create_fcontext_stack(size_t size);
void destroy_fcontext_stack(fcontext_stack_t *stack);
#ifdef __cplusplus
}
#endif
