// src/ItemCard.js
import React from 'react';
import './ItemCard.css';

function ItemCard({ item, onClick }) {
  return (
    <div className="item-card" onClick={() => onClick(item)}>
      <img src={item.imageUrl} alt={item.name} className="item-image" />
      <div className="item-info">
        <p><span>名称:</span> <span>{item.name}</span></p>
        <p><span>大小:</span> <span>{item.size}</span></p>
        <p><span>价格:</span> <span>{item.price}</span></p>
        <p><span>借出时间:</span> <span>{item.borrowedDate}</span></p>
        <p><span>归还日期:</span> <span>{item.returnDate}</span></p>
      </div>
    </div>
  );
}

export default ItemCard;
