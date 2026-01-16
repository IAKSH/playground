import { Box, Slider, Typography } from '@mui/material';

export default function SliderControl({
  label,
  value,
  onChange,
  onCommit,
  min = 0,
  max = 255,
  step = 1,
}) {
  return (
    <Box sx={{ px: 1 }}>
      <Typography variant="body2" gutterBottom>
        {label}: {value}
      </Typography>
      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(_, v) => onChange(v)}
        onChangeCommitted={(_, v) => onCommit?.(v)}  // 鼠标松开时触发
        valueLabelDisplay="auto"
      />
    </Box>
  );
}
