#include <assert.h>
#include <string.h>
#include "csi_output.h"
static uint8_t output[128];
static size_t used;
static int calls, waits, mode;
static int write_stub(const uint8_t *data, size_t size)
{
    ++calls;
    if (mode && calls <= 2) return calls == 1 ? 0 : -1;
    size_t n = mode && size > 3 ? 3 : size;
    memcpy(output + used, data, n);
    used += n;
    return n;
}
static void wait_stub(void) {++waits;}
int main(void)
{
    const uint8_t frame[] = {0xa5,'C','S','I',1,0,0,10,13,0,255};
    csi_write_complete(frame, sizeof(frame), write_stub, wait_stub);
    assert(calls == 1 && waits == 0 && used == sizeof(frame));
    assert(memcmp(output,frame,sizeof(frame))==0);
    used=0;calls=0;mode=1;
    csi_write_complete(frame, sizeof(frame), write_stub, wait_stub);
    csi_write_complete(frame, sizeof(frame), write_stub, wait_stub);
    assert(waits==2 && used==2*sizeof(frame));
    assert(memcmp(output,frame,sizeof(frame))==0);
    assert(memcmp(output+sizeof(frame),frame,sizeof(frame))==0);
    return 0;
}
