// Card.jsx
import React, { useState } from 'react';

const Card = (props) => {
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event) => {
    setRotation({
      x: -(event.nativeEvent.offsetY - event.target.offsetHeight / 2) / 7.5,
      y: (event.nativeEvent.offsetX - event.target.offsetWidth / 2) / 7.5,
    });
  };

  const handleMouseLeave = () => {
    setRotation({ x: 0, y: 0 });
  };

  return (
    <div
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        width: '200px',
        height: '200px',
        borderRadius: '10px',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
        transition: 'transform 0.5s',
        willChange: 'transform',
        transform: `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg)`,
      }}
    >
      {props.children}
    </div>
  );
};

export default Card;
