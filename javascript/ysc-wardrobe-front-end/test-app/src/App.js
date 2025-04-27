import React, { useState } from 'react';
import { Box, Typography } from '@mui/material';
import { styled } from '@mui/system';
import ClothGrid from './components/ClothGrid';
import backgroundImage from './assets/yunxi.png'; // 引用本地背景图片

import hanfu1 from './assets/hanfu1.jpg';
import hanfu2 from './assets/hanfu2.jpg';
import hanfu3 from './assets/hanfu3.jpg';
import hanfu4 from './assets/hanfu4.jpg';

const clothes = [
  { id: 1, pictureUrl: hanfu1, title: '汉服1' },
  { id: 2, pictureUrl: hanfu2, title: '汉服2' },
  { id: 3, pictureUrl: hanfu3, title: '汉服3' },
  { id: 4, pictureUrl: hanfu4, title: '汉服4' }
  // 在此处添加更多衣服对象
];

const BackgroundWrapper = styled(Box)({
  position: 'relative',
  padding: '20px',
  minHeight: '100vh',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundImage: `url(${backgroundImage})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    filter: 'blur(10px)',
    zIndex: -1,
  },
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(255, 255, 255, 0.1)',
    zIndex: -1,
  },
});

const StatusBar = styled(Box)({
  position: 'fixed',
  bottom: 0,
  left: 0,
  right: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  color: 'white',
  padding: '10px',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
});

const ScrollableContent = styled(Box)({
  overflowY: 'auto',
  height: 'calc(100vh - 60px)', // 计算减去状态栏的高度
  paddingBottom: '80px', // 确保最后一个元素不会被状态栏覆盖
});

function App() {
  const [currentClothID, setCurrentClothID] = useState(1);

  const handleTakeOut = async (id) => {
    setCurrentClothID(id);
    try {
      const response = await fetch(`/rotate/${id}`, {
        method: 'GET'
      });
      if (!response.ok) {
        throw new Error("Failed to call back-end rotate");
      }
    } catch (error) {
      console.error(error.message);
    }
  };  

  return (
    <BackgroundWrapper>
      <ScrollableContent>
        <ClothGrid clothes={clothes} currentClothID={currentClothID} onTakeOut={handleTakeOut} />
      </ScrollableContent>
      <StatusBar>
        <Typography>当前服装数量: {clothes.length}</Typography>
        <Typography>当前服装ID: {currentClothID}</Typography>
      </StatusBar>
    </BackgroundWrapper>
  );
}

export default App;
