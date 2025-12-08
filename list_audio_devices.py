import sounddevice as sd

print("Available audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"\nDevice {i}: {device['name']}")
    print(f"  Default sample rate: {device['default_samplerate']}")
    print(f"  Max input channels: {device['max_input_channels']}")
    print(f"  Max output channels: {device['max_output_channels']}")

default_input = sd.default.device[0]
default_output = sd.default.device[1]
print(f"\nDefault input device index: {default_input}")
print(f"Default output device index: {default_output}")