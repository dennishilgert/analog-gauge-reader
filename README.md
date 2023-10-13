# Analog gauge reader

Read the value of analog gauges with image processing

Based on [this project](https://github.com/intel-iot-devkit/python-cv-samples/tree/master/examples/analog-gauge-reader) by Intel 

## Image Processing

- Hough Circle Transform
- Hough Lines Transform

## Testing process

- gauge-1     Success
- gauge-2     Failed (no filtered lines)
- gauge-3     Failed (no filtered lines - img resolution too high?)
- gauge-4     Failed (glass reflection missleading line)
- gauge-5     Failed (no filtered lines)
- gauge-6     Failed (no filtered lines - img resolution too high?)
- gauge-7     Failed (no filtered lines - img resolution too high?)
- gauge-8     Failed (no filtered lines - img resolution too high!)
- gauge-9     Failed (no filtered lines - img resolution too high?)
- gauge-10    Failed (glass reflection missleading line)
- gauge-11    Success (too noisy background and surroundings - long exec time - rel accurate)
- gauge-12    Failed (glass reflection missleading line)
- gauge-13    Failed (shadow on meter missleading line)
- gauge-14    Failed (unable to find circle)
- gauge-15    Failed (non optimal shape of meter - missleading line)
- gauge-16    Failed (no filtered lines - img resolution too high?)
- gauge-17    Failed (no filtered lines??)
- gauge-18    Success
- gauge-19    Failed (no filtered lines - too noisy background and surroundings)
- gauge-20    Success (optimized meter color manually)
- gauge-21    Success (optimized meter center manually)
- gauge-22    Success (optimized meter color manually)
- gauge-23    Success
