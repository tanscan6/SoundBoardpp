#pragma once

#ifdef __cplusplus
extern "C" {
#endif

bool InitVBCablePlayback();
bool PlaySoundToVBCable(const wchar_t* filePath);
bool IsSoundPlaying();
void StopVBCablePlayback();

#ifdef __cplusplus
}
#endif
