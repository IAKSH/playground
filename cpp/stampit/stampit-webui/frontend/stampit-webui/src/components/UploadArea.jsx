import { useRef } from 'react';
import { Box, Button, Typography } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

export default function UploadArea({ onFileSelected }) {
  const inputRef = useRef(null);

  return (
    <Box sx={{ p: 2, border: '1px dashed', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle1" gutterBottom>
        上传图片
      </Typography>
      <div className="toolbar">
        <Button
          variant="contained"
          startIcon={<CloudUploadIcon />}
          onClick={() => inputRef.current?.click()}
        >
          选择文件
        </Button>
        <Typography variant="body2" color="text.secondary">
          支持 PNG/JPG/JPEG/BMP/GIF
        </Typography>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelected(file);
        }}
      />
    </Box>
  );
}
