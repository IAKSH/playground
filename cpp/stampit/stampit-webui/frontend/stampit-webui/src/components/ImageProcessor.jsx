import { useEffect, useState } from 'react';
import {
  Box, Grid, Paper, Typography, Divider,
  Backdrop, CircularProgress, Alert, Snackbar, Button
} from '@mui/material';
import UploadArea from './UploadArea';
import SliderControl from './SliderControl';
import ImageViewer from './ImageViewer';
import CropToolbar from './CropToolbar';
import {
  uploadImage,
  processImage,
  getThresholdedPng,
  getFinalPng,
  getBoundary,
  saveCrop,
  getStatus,
  clearSessionId, // <- ensure api exports this
} from '../api';

export default function ImageProcessor() {
  const [threshold, setThreshold] = useState(128);
  const [minArea, setMinArea] = useState(50);
  const [margin, setMargin] = useState(10);

  const [thresholdedUrl, setThresholdedUrl] = useState(null);
  const [finalUrl, setFinalUrl] = useState(null);
  const [boundary, setBoundary] = useState(null);
  const [selection, setSelection] = useState(null);

  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState(null);

  // 新增：会话过期与错误提示状态
  const [sessionExpired, setSessionExpired] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [snackOpen, setSnackOpen] = useState(false);

  const refreshStatus = async () => {
    try {
      const s = await getStatus();
      setInfo(s);
    } catch (e) {
      // 如果是会话错误，统一处理
      if (e.isSessionError) {
        handleSessionExpired(e);
      } else {
        console.error(e);
      }
    }
  };

  useEffect(() => {
    refreshStatus();
  }, []);

  // 统一会话过期处理
  const handleSessionExpired = (err) => {
    // 清理本地 session id（api 模块应提供）
    try { clearSessionId(); } catch (_) {}
    // 清理 UI 状态
    setThresholdedUrl(null);
    setFinalUrl(null);
    setBoundary(null);
    setSelection(null);
    setInfo(null);

    // 显示提示
    setSessionExpired(true);
    setErrorMsg(err?.message || '会话不存在或已过期，请重新上传图片。');
    setSnackOpen(true);
  };

  const handleUpload = async (file) => {
    setBusy(true);
    try {
      await uploadImage(file);
      await refreshStatus();
      await handleProcess();   // 上传成功后自动处理
    } catch (e) {
      // 如果是会话错误（极少见：上传时传了错误 sessionId），也处理
      if (e.isSessionError) {
        handleSessionExpired(e);
      } else {
        alert(`上传失败: ${e.message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const handleProcess = async () => {
    setBusy(true);
    try {
      const res = await processImage({ threshold, min_area: minArea, margin });
      setBoundary(res.boundary);
      setSelection(res.boundary);
      const thUrl = await getThresholdedPng();
      const finUrl = await getFinalPng();
      setThresholdedUrl(thUrl);
      setFinalUrl(finUrl);
      await refreshStatus();
    } catch (e) {
      if (e.isSessionError) {
        handleSessionExpired(e);
      } else {
        alert(`处理失败: ${e.message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const handleSaveCrop = async ({ x, y, w, h }) => {
    setBusy(true);
    try {
      await saveCrop({ x, y, w, h });
      // 可显示成功提示
    } catch (e) {
      if (e.isSessionError) {
        handleSessionExpired(e);
      } else {
        alert(`保存失败: ${e.message}`);
      }
    } finally {
      setBusy(false);
    }
  };

  // 用户点击“重新上传/清除会话”按钮
  const handleResetSession = () => {
    clearSessionId();
    setSessionExpired(false);
    setSnackOpen(false);
    setErrorMsg(null);
    setThresholdedUrl(null);
    setFinalUrl(null);
    setBoundary(null);
    setSelection(null);
    setInfo(null);
  };

  return (
    <Box sx={{ p: { xs: 1, md: 2 }, position: 'relative' }}>
      {/* 全局加载遮罩 */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={busy}
      >
        <CircularProgress color="inherit" />
      </Backdrop>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>控制面板</Typography>
            <UploadArea onFileSelected={handleUpload} />
            <Divider sx={{ my: 2 }} />
            <SliderControl
              label="阈值"
              value={threshold}
              onChange={setThreshold}
              onCommit={() => handleProcess()}
              min={0}
              max={255}
            />
            <SliderControl
              label="降噪强度"
              value={minArea}
              onChange={setMinArea}
              onCommit={() => handleProcess()}
              min={0}
              max={1000}
              step={10}
            />
            <SliderControl
              label="边界外扩"
              value={margin}
              onChange={setMargin}
              onCommit={() => handleProcess()}
              min={0}
              max={100}
            />
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                {info ? `状态：原图=${info.has_original}, 已处理=${info.has_processed}` : '状态未知'}
              </Typography>
            </Box>

            {/* 会话过期提示区域（可选，显式显示） */}
            {sessionExpired && (
              <Box sx={{ mt: 2 }}>
                <Alert severity="warning"
                  action={
                    <Button color="inherit" size="small" onClick={handleResetSession}>
                      重新开始
                    </Button>
                  }
                >
                  会话不存在或已过期，请重新上传图片。
                </Alert>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <ImageViewer
              thresholdedUrl={thresholdedUrl}
              finalUrl={finalUrl}
              boundary={boundary}
              onSelectionChange={setSelection}
            />
            <CropToolbar
              selection={selection}
              onSave={handleSaveCrop}
              hasImage={!!finalUrl}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* 全局 Snackbar 提示（可替代上方 Alert） */}
      <Snackbar
        open={snackOpen}
        autoHideDuration={8000}
        onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackOpen(false)}
          severity="warning"
          sx={{ width: '100%' }}
          action={
            <Button color="inherit" size="small" onClick={handleResetSession}>
              重新开始
            </Button>
          }
        >
          {errorMsg || '会话不存在或已过期，请重新上传图片。'}
        </Alert>
      </Snackbar>
    </Box>
  );
}
