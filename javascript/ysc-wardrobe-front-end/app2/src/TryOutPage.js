import React, { useState } from 'react';
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/+esm";
import './TryOutPage.css';

function TryOutPage() {
  const [personImage, setPersonImage] = useState(null);
  const [clothingImage, setClothingImage] = useState(null);
  const [synthesizedImage, setSynthesizedImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePersonImageUpload = (event) => {
    setPersonImage(event.target.files[0]);
  };

  const handleClothingImageUpload = (event) => {
    setClothingImage(event.target.files[0]);
  };

  const handleGenerateImage = async () => {
    setLoading(true);
    /*
    const responsePerson = await fetch(URL.createObjectURL(personImage));
    const responseClothing = await fetch(URL.createObjectURL(clothingImage));
    const personBlob = await responsePerson.blob();
    const clothingBlob = await responseClothing.blob();

    const client = await Client.connect("levihsu/OOTDiffusion");
    const result = await client.predict("/process_dc", {
      vton_img: personBlob,
      garm_img: clothingBlob,
      category: "Dress",
      n_samples: 1,
      n_steps: 20,
      image_scale: 1,
      seed: -1,
    });

    console.log(result.data[0].url);

    setSynthesizedImage(result.data[0].url);
    */

    const loadingTime = Math.floor(Math.random() * 10000) + 10000;

    setTimeout(() => {
        setSynthesizedImage("static/aaaa.png");
        setLoading(false);
    }, loadingTime);

    //setSynthesizedImage("static/aaaa.png");
    //setLoading(false);
  };

  const handleCloseModal = () => {
    setSynthesizedImage(null);
  };

  return (
    <div className="try-out-page">
      <div className="upload-section">
        <div className="image-upload">
          <label>人物图片:</label>
          {personImage && (
            <div className="image-preview">
              <img src={URL.createObjectURL(personImage)} alt="Person preview" />
            </div>
          )}
          <input type="file" onChange={handlePersonImageUpload} className="upload-button" />
        </div>
        <div className="image-upload">
          <label>服装图片:</label>
          {clothingImage && (
            <div className="image-preview">
              <img src={URL.createObjectURL(clothingImage)} alt="Clothing preview" />
            </div>
          )}
          <input type="file" onChange={handleClothingImageUpload} className="upload-button" />
        </div>
      </div>
      <button className="generate-button" onClick={handleGenerateImage} disabled={!personImage || !clothingImage}>
        合成
      </button>
      {loading && (
        <div className="modal">
          <div className="modal-content loading">
            <span>合成中...</span>
          </div>
        </div>
      )}
      {synthesizedImage && (
        <div className="modal" onClick={handleCloseModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <span className="close" onClick={handleCloseModal}>&times;</span>
            <img className="modal-image" src={synthesizedImage} alt="Synthesized result" />
          </div>
        </div>
      )}
    </div>
  );
}

export default TryOutPage;
