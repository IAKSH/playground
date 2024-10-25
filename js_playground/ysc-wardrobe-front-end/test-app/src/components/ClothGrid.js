// ClothGrid.js
import React, { useState } from 'react';
import { Box } from '@mui/material';
import { styled } from '@mui/system';
import ClothCard from './ClothCard';
import ClothDialog from './ClothDialog';

const GridWrapper = styled(Box)({
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
  gap: '20px',
});

const ClothGrid = ({ clothes, currentClothID, onTakeOut }) => {
  const [selectedCloth, setSelectedCloth] = useState(null);

  const handleOpenDialog = (cloth) => {
    setSelectedCloth(cloth);
  };

  const handleCloseDialog = () => {
    setSelectedCloth(null);
  };

  const handleTakeOut = () => {
    if (selectedCloth) {
      onTakeOut(selectedCloth.id);
      setSelectedCloth(null);
    }
  };

  return (
    <>
      <GridWrapper>
        {clothes.map((cloth) => (
          <ClothCard
            key={cloth.id}
            cloth={cloth}
            onClick={handleOpenDialog}
            selected={cloth.id === currentClothID} // 动态选择卡片
          />
        ))}
      </GridWrapper>
      <ClothDialog cloth={selectedCloth} onClose={handleCloseDialog} onTakeOut={handleTakeOut} />
    </>
  );
};

export default ClothGrid;
