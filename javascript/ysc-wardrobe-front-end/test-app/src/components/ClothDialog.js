// ClothDialog.js
import React from 'react';
import { Dialog, DialogActions, DialogContent, DialogTitle, Button } from '@mui/material';

const ClothDialog = ({ cloth, onClose, onTakeOut }) => (
  <Dialog open={!!cloth} onClose={onClose}>
    {cloth && (
      <>
        <DialogTitle>{cloth.title}</DialogTitle>
        <DialogContent>
          <img src={cloth.pictureUrl} alt={cloth.title} style={{ width: '100%' }} />
        </DialogContent>
        <DialogActions>
          <Button onClick={onTakeOut} color="primary">取出</Button>
        </DialogActions>
      </>
    )}
  </Dialog>
);

export default ClothDialog;
