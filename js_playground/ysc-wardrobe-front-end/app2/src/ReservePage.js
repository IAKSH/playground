import React, { useState } from 'react';
import ItemCard from './ItemCard';
import './ReservePage.css';

function ReservePage() {
  const [items, setItems] = useState([
    {
      name: '凤翅紫金冠',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    },
    {
      name: '锁子黄金甲',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
      imageUrl: 'logo192.png'
    },
    {
      name: '藕丝步云履',
      size: 'L',
      price: '￥200',
      borrowedDate: 'N/A',
      returnDate: 'N/A',
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

  const handleTakeOut = () => {
    console.log('立即取衣', selectedItem);
    if (selectedItem) {
      const utterance = new SpeechSynthesisUtterance(`请取出 ${selectedItem.name}`);
      // 列出所有可用的声音
      const voices = window.speechSynthesis.getVoices();
      //console.log(voices);
      // 选择一个声音
      utterance.voice = voices[0];
      window.speechSynthesis.speak(utterance);
    }
    // TODO
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
          <p style={{ color: 'black' }}>从隔壁复制过来的页面，不知道要显示什么</p>
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
              <button className="return-button" onClick={handleTakeOut}>立即取衣</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ReservePage;
