/* WebAssembly port, 2026-10-17. NetHack General Public License; see licenses/NetHack.txt. */
/* One NLE instance per Web Worker; a restart creates fresh engine globals. */
#include "hack.h"
#include "nleobs.h"
typedef struct nle_globals nle_ctx_t;
nle_ctx_t *nle_start(nle_obs *, FILE *, nle_seeds_init_t *, nle_settings *);
nle_ctx_t *nle_step(nle_ctx_t *, nle_obs *);
#include <stdint.h>
#include <string.h>
static nle_obs obs;
static nle_ctx_t *ctx;
static short glyphs[21*79], inv_glyphs[55];
static unsigned char chars[21*79], colors[21*79], message[256];
static unsigned char inv_strs[55*80], inv_letters[55], inv_oclasses[55];
static long blstats[27];
static int internal[9], game_program_state[6], misc[3];
int wand_start(unsigned int seed, const char *options) {
    nle_settings config = {0};
    strcpy(config.hackdir, "/game");
    strncpy(config.options, options, sizeof(config.options)-1);
    config.spawn_monsters = 0;
    nle_seeds_init_t seeds = {{seed,seed},0};
    obs.glyphs=glyphs; obs.chars=chars; obs.colors=colors; obs.message=message;
    obs.blstats=blstats; obs.internal=internal; obs.program_state=game_program_state; obs.misc=misc;
    obs.inv_glyphs=inv_glyphs; obs.inv_letters=inv_letters; obs.inv_oclasses=inv_oclasses; obs.inv_strs=inv_strs;
    ctx=nle_start(&obs, NULL, &seeds, &config);
    return obs.done;
}
int wand_step(int key) {
    obs.action=key;
    ctx=nle_step(ctx,&obs);
    return obs.done;
}
void *wand_pointer(int field) {
    switch(field) {
    case 0:return glyphs; case 1:return chars; case 2:return colors; case 3:return blstats;
    case 4:return message; case 5:return inv_strs; case 6:return inv_letters;
    case 7:return internal; case 8:return game_program_state; case 9:return misc;
    case 10:return inv_glyphs; case 11:return inv_oclasses;
    default:return NULL;
    }
}
int wand_flag(int field) {
    return field==0 ? obs.done : field==1 ? obs.in_normal_game : obs.how_done;
}
