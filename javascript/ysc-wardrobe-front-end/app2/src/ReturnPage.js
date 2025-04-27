import React, { useState } from 'react';
import ItemCard from './ItemCard';
import './ReturnPage.css';

function ReturnPage() {
  const [items, setItems] = useState([
    {
      name: '汉服A',
      size: 'L',
      price: '￥200',
      borrowedDate: '2024-01-01',
      returnDate: '2024-01-10',
      image: 'static/b1.jpg'
    },
    {
      name: '汉服B',
      size: 'L',
      price: '￥200',
      borrowedDate: '2024-01-01',
      returnDate: '2024-01-10',
      image: 'static/c2.jpg'
    }
    // 添加更多项...
  ]);

  const [selectedItem, setSelectedItem] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [isBrokenSuccess, setIsBrokenSuccess] = useState(false);

  const handleClick = (item) => {
    setSelectedItem(item);
  };

  const handleClose = () => {
    setSelectedItem(null);
  };

  const handleReturn = () => {
    console.log('立即归还', selectedItem);
    setIsLoading(true);

    const loadingTime = Math.floor(Math.random() * 1000) + 500;

    setTimeout(() => {
      setIsLoading(false);
      setItems(items.filter(item => item !== selectedItem));
      setSelectedItem(null);
      setIsSuccess(true);

      handleClose();

      setTimeout(() => {
        setIsSuccess(false);
      }, 1000);
    }, loadingTime);
  };

  const handleDamage = () => {
    console.log('衣物破损', selectedItem);
    setIsLoading(true);

    const loadingTime = Math.floor(Math.random() * 1000) + 500;

    setTimeout(() => {
      setIsLoading(false);
      setItems(items.filter(item => item !== selectedItem));
      setSelectedItem(null);
      setIsBrokenSuccess(true);

      handleClose();

      setTimeout(() => {
        setIsBrokenSuccess(false);
      }, 1000);
    }, loadingTime);

    handleClose();
  };

  return (
    <div className="return-page">
      <div className="content-container">
        <div className="items-container">
          {items.map((item, index) => (
            <ItemCard key={index} item={item} onClick={() => handleClick(item)} />
          ))}
        </div>
      </div>

      {selectedItem && (
        <div className="modal-overlay" onClick={handleClose}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <img src={selectedItem.image} alt={selectedItem.name} className="modal-image" />
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

      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-content">连接中...</div>
        </div>
      )}

      {isSuccess && (
        <div className="success-dialog">
          归还成功
        </div>
      )}

      {isBrokenSuccess && (
        <div className="broken-success-dialog">
          破损已记录
        </div>
      )}
    </div>
  );
}

export default ReturnPage;
