// ClothCard.js
import React from 'react';
import { Card, CardMedia, CardContent, Typography } from '@mui/material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)(({ selected }) => ({
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'space-between',
  opacity: 0.9, // 半透明效果
  transition: 'transform 0.3s, opacity 0.3s', // 动画效果
  border: selected ? '2px solid blue' : 'none', // 蓝色光效
  '&:hover': {
    transform: 'scale(1.05) rotateY(5deg)', // 3D效果
    opacity: 1, // 鼠标悬停时变为不透明
  }
}));

const CardImage = styled(CardMedia)({
  height: '350px', // 设置图片的固定高度
  objectFit: 'cover', // 确保图片按比例填充
});

const CardText = styled(CardContent)({
  flexGrow: 0, // 确保文本内容不会被挤压
});

const ClothCard = ({ cloth, onClick, selected }) => (
  <StyledCard onClick={() => onClick(cloth)} selected={selected}>
    <CardImage component="img" image={cloth.pictureUrl} title={cloth.title} />
    <CardText>
      <Typography variant="h6">{cloth.title}</Typography>
    </CardText>
  </StyledCard>
);

export default ClothCard;
