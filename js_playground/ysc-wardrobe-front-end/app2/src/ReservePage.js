import React, { useState } from 'react';
import ItemCard from './ItemCard';
import './ReservePage.css';

function ReservePage() {
  const [items, setItems] = useState([
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: '2024-01-01',
      returnDate: '2024-01-10',
      imageUrl: 'logo192.png'
    },
    {
        name: '汉服A',
        size: 'L',
        price: '￥200',
        borrowedDate: '2024-01-01',
        returnDate: '2024-01-10',
        imageUrl: 'logo192.png'
      }
    // 添加更多项...
  ]);

  const [selectedItem, setSelectedItem] = useState(null);

  const handleClick = (item) => {
    setSelectedItem(item);
  };

  const handleClose = () => {
    setSelectedItem(null);
  };

  const handleReturn = () => {
    console.log('立即归还', selectedItem);
    // 在这里添加处理立即归还的逻辑
    handleClose();
  };

  const handleDamage = () => {
    console.log('衣物破损', selectedItem);
    // 在这里添加处理衣物破损的逻辑
    handleClose();
  };

  return (
    <div className="return-page">
      <div className="content-container">
        <div className="items-container">
          {items.map((item, index) => (
            <ItemCard key={index} item={item} onClick={handleClick} />
          ))}
        </div>
        <div className="button-container">
          <p style={{ color:'black' }}>从隔壁复制过来的页面，不知道要显示什么</p>
        </div>
      </div>

      {selectedItem && (
        <div className="modal-overlay" onClick={handleClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <img src={selectedItem.imageUrl} alt={selectedItem.name} className="modal-image" />
            <div className="modal-info">
              <p><span>名称:</span> <span>{selectedItem.name}</span></p>
              <p><span>大小:</span> <span>{selectedItem.size}</span></p>
              <p><span>价格:</span> <span>{selectedItem.price}</span></p>
              <p><span>借出时间:</span> <span>{selectedItem.borrowedDate}</span></p>
              <p><span>归还日期:</span> <span>{selectedItem.returnDate}</span></p>
            </div>
            <div className="modal-actions">
              <button className="return-button" onClick={handleReturn}>立即归还</button>
              <button className="damage-button" onClick={handleDamage}>衣物破损</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ReservePage;
