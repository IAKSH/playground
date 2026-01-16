import React from 'react';
import { Box, Button } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';

export default function CropToolbar({ selection, onSave, hasImage }) {
  return (
    <Box sx={{ mt: 2 }} className="toolbar">
      <Button
        variant="contained"
        startIcon={<SaveIcon />}
        disabled={!hasImage || !selection || selection.w <= 0 || selection.h <= 0}
        onClick={() => onSave?.(selection)}
      >
        下载
      </Button>
    </Box>
  );
}
