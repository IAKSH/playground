import React from 'react';
import './ItemCard.css';

function ItemCard({ item, onClick }) {
  return (
    <div className="item-card" onClick={() => onClick(item)}>
      <img src={item.image} alt={item.name} className="item-image" />
      <div className="item-info">
        <p><span>名称:</span> <span>{item.name}</span></p>
        <p><span>店铺:</span> <span>{item.shopName}</span></p>
        <p><span>价格:</span> <span>{item.price}</span></p>
      </div>
    </div>
  );
}

export default ItemCard;
