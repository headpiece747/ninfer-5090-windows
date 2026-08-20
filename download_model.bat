@echo off
echo =======================================================
echo Downloading Qwen3.8-27B NVFP4 Model...
echo File Size: ~20GB
echo Please wait, this may take a while depending on your network.
echo =======================================================
curl -L -C - -o qwen3_8_27b_nvfp4.ninfer https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/main/qwen3_8_27b_nvfp4.ninfer
echo =======================================================
echo Download complete! You can now start the server.
echo =======================================================
pause
