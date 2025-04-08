import streamlit as st
from clearvoice import ClearVoice
import os
import tempfile
import zipfile
import io
import time
import shutil

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")
temp_dir = 'temp'

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Check if temp directory exists, create if not
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Save to temp directory, overwrite if file exists
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return temp_path
    return None

def save_uploaded_files(uploaded_files):
    file_paths = []
    if uploaded_files:
        # Check if temp directory exists, create if not
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        for uploaded_file in uploaded_files:
            # Save to temp directory, overwrite if file exists
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            file_paths.append(temp_path)
    return file_paths

def create_download_zip(directory):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)
    memory_file.seek(0)
    return memory_file

def main():
    st.title("ClearerVoice Studio")
    
    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction"])
    
    with tabs[0]:
        st.header("Speech Enhancement")
        
        # Model selection
        se_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        selected_model = st.selectbox("Select Model", se_models)
        
        # File upload - now supports multiple files
        uploaded_files = st.file_uploader("Upload Audio Files", type=['wav'], key='se', accept_multiple_files=True)
        
        bulk_processing = st.checkbox("Enable Bulk Processing", key='se_bulk')
        
        if st.button("Start Processing", key='se_process'):
            if uploaded_files:
                # Create a unique session output directory
                session_id = int(time.time())
                output_dir = os.path.join(temp_dir, f"speech_enhancement_output_{session_id}")
                os.makedirs(output_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process files one by one
                with st.spinner('Processing files...'):
                    if bulk_processing:
                        # Save all uploaded files
                        input_paths = save_uploaded_files(uploaded_files)
                        
                        # Initialize ClearVoice once for all files
                        myClearVoice = ClearVoice(task='speech_enhancement', 
                                                model_names=[selected_model])
                        
                        for i, input_path in enumerate(input_paths):
                            status_text.text(f"Processing file {i+1}/{len(input_paths)}: {os.path.basename(input_path)}")
                            progress_bar.progress((i) / len(input_paths))
                            
                            # Process audio
                            output_wav = myClearVoice(input_path=input_path, 
                                                    online_write=False)
                            
                            # Save processed audio
                            file_name = os.path.basename(input_path).split('.')[0]
                            output_path = os.path.join(output_dir, f"output_{selected_model}_{file_name}.wav")
                            myClearVoice.write(output_wav, output_path=output_path)
                        
                        progress_bar.progress(1.0)
                        status_text.text(f"Completed processing {len(input_paths)} files")
                        
                        # Create zip file for download
                        zip_buffer = create_download_zip(output_dir)
                        
                        # Offer download button
                        st.download_button(
                            label="Download All Processed Files",
                            data=zip_buffer,
                            file_name=f"speech_enhancement_results_{session_id}.zip",
                            mime="application/zip"
                        )
                    else:
                        # Process only the first file for single file mode
                        input_path = save_uploaded_file(uploaded_files[0])
                        
                        # Initialize ClearVoice
                        myClearVoice = ClearVoice(task='speech_enhancement', 
                                                model_names=[selected_model])
                        
                        # Process audio
                        output_wav = myClearVoice(input_path=input_path, 
                                                online_write=False)
                        
                        # Save processed audio
                        file_name = os.path.basename(input_path).split('.')[0]
                        output_path = os.path.join(output_dir, f"output_{selected_model}_{file_name}.wav")
                        myClearVoice.write(output_wav, output_path=output_path)
                        
                        # Display audio
                        st.audio(output_path)
                        
                        # Offer download button for single file
                        with open(output_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Processed File",
                                data=file,
                                file_name=os.path.basename(output_path),
                                mime="audio/wav"
                            )
            else:
                st.error("Please upload at least one audio file")
    
    with tabs[1]:
        st.header("Speech Separation")
        
        # File upload - now supports multiple files
        uploaded_files = st.file_uploader("Upload Mixed Audio Files", type=['wav', 'avi'], key='ss', accept_multiple_files=True)
        
        bulk_processing = st.checkbox("Enable Bulk Processing", key='ss_bulk')
        
        if st.button("Start Separation", key='ss_process'):
            if uploaded_files:
                # Create a unique session output directory
                session_id = int(time.time())
                output_dir = os.path.join(temp_dir, f"speech_separation_output_{session_id}")
                os.makedirs(output_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner('Processing files...'):
                    if bulk_processing:
                        # Save all uploaded files
                        input_paths = save_uploaded_files(uploaded_files)
                        
                        # Initialize ClearVoice once for all files
                        myClearVoice = ClearVoice(task='speech_separation', 
                                                model_names=['MossFormer2_SS_16K'])
                        
                        for i, input_path in enumerate(input_paths):
                            status_text.text(f"Processing file {i+1}/{len(input_paths)}: {os.path.basename(input_path)}")
                            progress_bar.progress((i) / len(input_paths))
                            
                            # Extract audio if input is video file
                            if input_path.endswith(('.avi')):
                                import cv2
                                video = cv2.VideoCapture(input_path)
                                audio_path = input_path.replace('.avi','.wav')
                                
                                # Extract audio
                                import subprocess
                                cmd = f"ffmpeg -i {input_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                                subprocess.call(cmd, shell=True)
                                
                                input_path = audio_path
                            
                            # Process audio
                            output_wav = myClearVoice(input_path=input_path, 
                                                    online_write=False)
                            
                            file_name = os.path.basename(input_path).split('.')[0]
                            base_file_name = 'output_MossFormer2_SS_16K_'
                            
                            # Save processed audio
                            output_path = os.path.join(output_dir, f"{base_file_name}{file_name}.wav")
                            myClearVoice.write(output_wav, output_path=output_path)
                        
                        progress_bar.progress(1.0)
                        status_text.text(f"Completed processing {len(input_paths)} files")
                        
                        # Create zip file for download
                        zip_buffer = create_download_zip(output_dir)
                        
                        # Offer download button
                        st.download_button(
                            label="Download All Processed Files",
                            data=zip_buffer,
                            file_name=f"speech_separation_results_{session_id}.zip",
                            mime="application/zip"
                        )
                    else:
                        # Process only the first file for single file mode
                        input_path = save_uploaded_file(uploaded_files[0])

                        # Extract audio if input is video file
                        if input_path.endswith(('.avi')):
                            import cv2
                            video = cv2.VideoCapture(input_path)
                            audio_path = input_path.replace('.avi','.wav')
                            
                            # Extract audio
                            import subprocess
                            cmd = f"ffmpeg -i {input_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                            subprocess.call(cmd, shell=True)
                            
                            input_path = audio_path
                        
                        # Initialize ClearVoice
                        myClearVoice = ClearVoice(task='speech_separation', 
                                                model_names=['MossFormer2_SS_16K'])
                        
                        # Process audio
                        output_wav = myClearVoice(input_path=input_path, 
                                                online_write=False)
                        
                        file_name = os.path.basename(input_path).split('.')[0]
                        base_file_name = 'output_MossFormer2_SS_16K_'
                        
                        # Save processed audio
                        output_path = os.path.join(output_dir, f"{base_file_name}{file_name}.wav")
                        myClearVoice.write(output_wav, output_path=output_path)
                        
                        # Display audio
                        st.audio(output_path)
                        
                        # Offer download button for single file
                        with open(output_path, "rb") as file:
                            btn = st.download_button(
                                label="Download Processed File",
                                data=file,
                                file_name=os.path.basename(output_path),
                                mime="audio/wav"
                            )
            else:
                st.error("Please upload at least one audio file")
    
    with tabs[2]:
        st.header("Target Speaker Extraction")
        
        # File upload - now supports multiple files
        uploaded_files = st.file_uploader("Upload Video Files", type=['mp4', 'avi'], key='tse', accept_multiple_files=True)
        
        bulk_processing = st.checkbox("Enable Bulk Processing", key='tse_bulk')
        
        if st.button("Start Extraction", key='tse_process'):
            if uploaded_files:
                # Create a unique session output directory
                session_id = int(time.time())
                output_dir = os.path.join(temp_dir, f"videos_tse_output_{session_id}")
                os.makedirs(output_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner('Processing files...'):
                    if bulk_processing:
                        # Save all uploaded files
                        input_paths = save_uploaded_files(uploaded_files)
                        
                        # Initialize ClearVoice once for all files
                        myClearVoice = ClearVoice(task='target_speaker_extraction', 
                                                model_names=['AV_MossFormer2_TSE_16K'])
                        
                        for i, input_path in enumerate(input_paths):
                            status_text.text(f"Processing file {i+1}/{len(input_paths)}: {os.path.basename(input_path)}")
                            progress_bar.progress((i) / len(input_paths))
                            
                            # Create file-specific output directory
                            file_name = os.path.basename(input_path).split('.')[0]
                            file_output_dir = os.path.join(output_dir, file_name)
                            os.makedirs(file_output_dir, exist_ok=True)
                            
                            # Process video
                            myClearVoice(input_path=input_path, 
                                         online_write=True,
                                         output_path=file_output_dir)
                        
                        progress_bar.progress(1.0)
                        status_text.text(f"Completed processing {len(input_paths)} files")
                        
                        # Create zip file for download
                        zip_buffer = create_download_zip(output_dir)
                        
                        # Offer download button
                        st.download_button(
                            label="Download All Processed Files",
                            data=zip_buffer,
                            file_name=f"target_speaker_extraction_results_{session_id}.zip",
                            mime="application/zip"
                        )
                    else:
                        # Process only the first file for single file mode
                        input_path = save_uploaded_file(uploaded_files[0])
                        
                        # Initialize ClearVoice
                        myClearVoice = ClearVoice(task='target_speaker_extraction', 
                                                model_names=['AV_MossFormer2_TSE_16K'])
                        
                        # Process video
                        myClearVoice(input_path=input_path, 
                                     online_write=True,
                                     output_path=output_dir)
                        
                        # Display output folder
                        st.subheader("Output Folder")
                        st.text(output_dir)
                        
                        # Create zip file for download
                        zip_buffer = create_download_zip(output_dir)
                        
                        # Offer download button
                        st.download_button(
                            label="Download Processed Files",
                            data=zip_buffer,
                            file_name=f"target_speaker_extraction_results.zip",
                            mime="application/zip"
                        )
            else:
                st.error("Please upload at least one video file")

if __name__ == "__main__":    
    main()