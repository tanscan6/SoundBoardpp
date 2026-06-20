#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

static ma_decoder gDecoder{};
static ma_device gDevice{};
static bool gDeviceInitialized = false;
static bool gDecoderInitialized = false;
static std::atomic<bool> gPlaying{false};

static ma_device_id gVBCableId{};
static bool gVBCableFound = false;

static void data_callback(ma_device* device, void* output, const void* input, ma_uint32 frameCount)
{
    if (!gDecoderInitialized)
    {
        ma_uint32 bytesPerFrame = device->playback.channels * sizeof(float);
        memset(output, 0, frameCount * bytesPerFrame);
        return;
    }

    ma_uint64 readFrames = 0;
    ma_decoder_read_pcm_frames(&gDecoder, output, frameCount, &readFrames);

    if (readFrames == 0)
    {
        gPlaying = false;
        ma_uint32 bytesPerFrame = device->playback.channels * sizeof(float);
        memset(output, 0, frameCount * bytesPerFrame);
    }
}

extern "C" {

bool InitVBCablePlayback()
{
    if (gVBCableFound) return true;

    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS)
        return false;

    ma_device_info* playbackDevices = nullptr;
    ma_uint32 playbackDeviceCount = 0;
    ma_context_get_devices(&context, &playbackDevices, &playbackDeviceCount, nullptr, nullptr);

    for (ma_uint32 i = 0; i < playbackDeviceCount; i++)
    {
        const char* name = playbackDevices[i].name;
        if (strstr(name, "CABLE") || strstr(name, "VB-Audio") || strstr(name, "Virtual"))
        {
            gVBCableId = playbackDevices[i].id;
            gVBCableFound = true;
            break;
        }
    }

    ma_context_uninit(&context);
    return gVBCableFound;
}

bool PlaySoundToVBCable(const wchar_t* filePath)
{
    if (!InitVBCablePlayback())
        return false;

    if (gDeviceInitialized)
    {
        ma_device_uninit(&gDevice);
        gDeviceInitialized = false;
    }
    if (gDecoderInitialized)
    {
        ma_decoder_uninit(&gDecoder);
        gDecoderInitialized = false;
    }

    ma_result res = ma_decoder_init_file_w(filePath, nullptr, &gDecoder);
    if (res != MA_SUCCESS)
    {
        ma_decoder_uninit(&gDecoder);
        gDecoderInitialized = false;
        return false;
    }
    gDecoderInitialized = true;

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.pDeviceID = &gVBCableId;
    config.playback.format = gDecoder.outputFormat;
    config.playback.channels = gDecoder.outputChannels;
    config.sampleRate = gDecoder.outputSampleRate;
    config.dataCallback = data_callback;
    config.periodSizeInMilliseconds = 100;

    res = ma_device_init(nullptr, &config, &gDevice);
    if (res != MA_SUCCESS)
    {
        ma_decoder_uninit(&gDecoder);
        gDecoderInitialized = false;
        return false;
    }
    gDeviceInitialized = true;

    gPlaying = true;
    res = ma_device_start(&gDevice);
    if (res != MA_SUCCESS)
    {
        ma_device_uninit(&gDevice);
        gDeviceInitialized = false;
        ma_decoder_uninit(&gDecoder);
        gDecoderInitialized = false;
        return false;
    }

    return true;
}

bool IsSoundPlaying()
{
    return gPlaying.load();
}

void StopVBCablePlayback()
{
    if (gDeviceInitialized)
    {
        ma_device_uninit(&gDevice);
        gDeviceInitialized = false;
    }
    if (gDecoderInitialized)
    {
        ma_decoder_uninit(&gDecoder);
        gDecoderInitialized = false;
    }
    gPlaying = false;
}

}
