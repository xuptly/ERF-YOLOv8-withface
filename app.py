import streamlit as st
import os
import tempfile
from PIL import Image
import shutil
import sys
from pathlib import Path
import glob

# 页面配置
st.set_page_config(
    page_title="多任务环境感知系统",
    page_icon="🔍",
    layout="wide"
)

# 标题
st.title("🔍 多任务环境感知系统")
st.markdown("---")

# 初始化session状态
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'detected_images' not in st.session_state:
    st.session_state.detected_images = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# 环境诊断信息
with st.sidebar:
    st.header("🔧 系统状态")
    
    # 检查模型文件
    onnx_model_path = "best.onnx"
    if os.path.exists(onnx_model_path):
        file_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        st.success(f"✅ ONNX模型就绪")
        st.write(f"文件大小: {file_size:.1f} MB")
    else:
        st.error(f"❌ ONNX模型未找到")
        st.info("请确保 best.onnx 在根目录")

# 图片上传
uploaded_files = st.file_uploader(
    "选择一张或多张图片",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    accept_multiple_files=True,
    key="file_uploader"
)

# 保存上传的图片到临时目录
if uploaded_files:
    st.session_state.uploaded_images = []
    for uploaded_file in uploaded_files:
        try:
            # 保存图片到临时目录
            temp_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 将图片添加到session_state
            image = Image.open(uploaded_file)
            st.session_state.uploaded_images.append({
                'name': uploaded_file.name,
                'path': temp_path,
                'image': image
            })
        except Exception as e:
            st.error(f"处理图片 {uploaded_file.name} 时出错: {str(e)}")

# 显示上传的图片
if st.session_state.uploaded_images:
    st.subheader("📁 已上传的图片")

    # 显示图片网格
    cols = st.columns(min(3, len(st.session_state.uploaded_images)))
    for i, img_info in enumerate(st.session_state.uploaded_images):
        with cols[i % len(cols)]:
            st.image(img_info['image'], caption=img_info['name'], use_column_width=True)

    st.markdown("---")

    # 检测设置
    st.subheader("⚙️ 检测设置")
    col1, col2, col3 = st.columns(3)

    with col1:
        conf_threshold = st.slider("置信度阈值", 0.1, 1.0, 0.25, 0.05)
    with col2:
        iou_threshold = st.slider("IOU阈值", 0.1, 1.0, 0.45, 0.05)
    with col3:
        max_images = st.slider("最大处理数量", 1, 10, 3, 1)

    # 检测按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 开始检测", use_container_width=True, type="primary"):
            try:
                from ultralytics import YOLO

                # 使用 ONNX 模型
                model_path = "best.onnx"
                
                # 文件存在性检查
                if not os.path.exists(model_path):
                    st.error(f"❌ ONNX模型文件不存在: {model_path}")
                    st.info("请确保 best.onnx 文件与 app.py 在同一目录下")
                    st.stop()

                # 加载 ONNX 模型
                with st.spinner("正在加载模型..."):
                    model = YOLO(model_path)
                st.success("✅ ONNX 模型加载成功")

                with st.spinner("正在检测中，请稍候..."):
                    st.session_state.detected_images = []

                    # 创建专门的输出目录
                    output_dir = os.path.join(st.session_state.temp_dir, "detection_results")
                    os.makedirs(output_dir, exist_ok=True)

                    # 进度显示
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 限制处理图片数量（避免超时）
                    images_to_process = st.session_state.uploaded_images[:max_images]
                    
                    for i, img_info in enumerate(images_to_process):
                        try:
                            # 更新进度
                            progress = i / len(images_to_process)
                            progress_bar.progress(progress)
                            status_text.text(f"正在处理: {img_info['name']} ({i + 1}/{len(images_to_process)})")

                            # 使用模型进行预测
                            results = model.predict(
                                source=img_info['path'],
                                imgsz=640,
                                device='cpu',  # ONNX 模型使用 CPU
                                conf=conf_threshold,
                                iou=iou_threshold,
                                save=True,
                                project=output_dir,
                                exist_ok=True,
                                show=False
                            )

                            # 查找保存的结果图片
                            result_files = []
                            search_patterns = [
                                os.path.join(output_dir, "predict*", "*.jpg"),
                                os.path.join(output_dir, "*", "*.jpg"),
                                os.path.join(output_dir, "*.jpg")
                            ]
                            
                            for pattern in search_patterns:
                                result_files.extend(glob.glob(pattern))
                                if result_files:
                                    break

                            # 找到对应的结果文件
                            result_found = False
                            original_stem = Path(img_info['name']).stem.lower()
                            
                            for result_file in result_files:
                                result_stem = Path(result_file).stem.lower()
                                if original_stem in result_stem:
                                    try:
                                        detected_img = Image.open(result_file)
                                        st.session_state.detected_images.append({
                                            'name': f"检测_{img_info['name']}",
                                            'image': detected_img,
                                            'result_path': result_file
                                        })
                                        result_found = True
                                        break
                                    except Exception as img_error:
                                        continue

                            if not result_found and result_files:
                                # 如果没有精确匹配，使用最新的结果文件
                                latest_file = max(result_files, key=os.path.getctime)
                                try:
                                    detected_img = Image.open(latest_file)
                                    st.session_state.detected_images.append({
                                        'name': f"检测_{img_info['name']}",
                                        'image': detected_img,
                                        'result_path': latest_file
                                    })
                                    result_found = True
                                except Exception as e:
                                    pass

                            if not result_found:
                                st.warning(f"未找到图片 {img_info['name']} 的检测结果")

                        except Exception as e:
                            st.error(f"处理图片 {img_info['name']} 时出错: {str(e)}")

                    # 完成进度
                    progress_bar.progress(1.0)
                    status_text.text("检测完成！")

                    if st.session_state.detected_images:
                        st.success(f"✅ 成功检测 {len(st.session_state.detected_images)} 张图片")
                    else:
                        st.warning("⚠️ 未生成任何检测结果")
                        st.info("""
                        可能的原因：
                        1. 置信度阈值设置过高
                        2. 图片中没有检测到目标
                        3. 可以尝试降低阈值重新检测
                        """)

            except ImportError as e:
                st.error(f"无法导入YOLO模型: {str(e)}")
                st.info("请在 requirements.txt 中添加 ultralytics>=8.0.0")
            except Exception as e:
                st.error(f"检测过程中出错: {str(e)}")
                st.write("错误详情:", type(e).__name__)

# 显示检测结果
if st.session_state.detected_images:
    st.markdown("---")
    st.subheader("📊 检测结果对比")

    for i in range(len(st.session_state.detected_images)):
        col1, col2 = st.columns(2)

        with col1:
            if i < len(st.session_state.uploaded_images):
                st.markdown(f"**原图:** {st.session_state.uploaded_images[i]['name']}")
                st.image(st.session_state.uploaded_images[i]['image'], use_column_width=True)

        with col2:
            st.markdown(f"**检测结果:** {st.session_state.detected_images[i]['name']}")
            st.image(st.session_state.detected_images[i]['image'], use_column_width=True)

            # 添加下载按钮
            if 'result_path' in st.session_state.detected_images[i]:
                with open(st.session_state.detected_images[i]['result_path'], "rb") as file:
                    st.download_button(
                        label="📥 下载检测结果",
                        data=file,
                        file_name=f"detected_{st.session_state.detected_images[i]['name']}",
                        mime="image/jpeg",
                        key=f"download_{i}"
                    )

        if i < len(st.session_state.detected_images) - 1:
            st.markdown("---")

# 侧边栏信息
with st.sidebar:
    st.header("ℹ️ 使用说明")
    st.markdown("""
    1. **上传图片**: 选择要检测的图片
    2. **调整参数**: 设置置信度和IOU阈值
    3. **开始检测**: 点击检测按钮运行模型
    4. **查看结果**: 左右对比查看检测效果
    5. **下载结果**: 点击下载按钮保存检测结果

    **支持格式:** JPG, JPEG, PNG, BMP
    **检测模型:** ERF-YOLOv8 (ONNX)
    """)

    if st.session_state.uploaded_images:
        st.info(f"📤 已上传 {len(st.session_state.uploaded_images)} 张图片")
    else:
        st.warning("📤 等待图片上传")

    if st.session_state.detected_images:
        st.success(f"🎯 已检测 {len(st.session_state.detected_images)} 张图片")

    # 清理按钮
    st.markdown("---")
    if st.button("🗑️ 清理临时文件"):
        if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
            try:
                shutil.rmtree(st.session_state.temp_dir)
                st.session_state.temp_dir = tempfile.mkdtemp()
                st.session_state.uploaded_images = []
                st.session_state.detected_images = []
                st.success("临时文件已清理")
                st.rerun()
            except Exception as e:
                st.error(f"清理失败: {str(e)}")

# 显示临时目录信息（调试用）
with st.sidebar:
    if st.checkbox("显示调试信息"):
        st.write("临时目录:", st.session_state.temp_dir)
        st.write("上传图片:", [img['name'] for img in st.session_state.uploaded_images])
        st.write("检测图片:", [img['name'] for img in st.session_state.detected_images])

# 应用启动说明
if not st.session_state.uploaded_images and not st.session_state.detected_images:
    st.info("👆 请上传一张或多张图片进行目标检测")

# 清理函数
def cleanup():
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass

# 注册清理函数
import atexit
atexit.register(cleanup)
