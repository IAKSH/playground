from gradio_client import Client, handle_file

client = Client("levihsu/OOTDiffusion")

result = client.predict(
  vton_img=handle_file('https://levihsu-ootdiffusion.hf.space/file=/tmp/gradio/aa9673ab8fa122b9c5cdccf326e5f6fc244bc89b/model_8.png'),
  garm_img=handle_file('https://i.pinimg.com/550x/96/dc/3e/96dc3e361c2b9ed3e3804d50f33bfc6f.jpg'),
  category="Dress",
  n_samples=1,
  n_steps=20,
  image_scale=2,
  seed=-1,
  api_name="/process_dc"
)

print(result)