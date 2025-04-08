import streamlit as st
from clearvoice import ClearVoice
import os
import tempfile
import zipfile
import io
import time
import shutil
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict

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

def run_benchmark(audio_file, chunk_sizes, task, model_names):
    """
    Run benchmark tests on audio chunks of specified sizes.
    
    Args:
        audio_file: Path to the audio file
        chunk_sizes: List of chunk sizes in milliseconds
        task: Task type ('speech_enhancement', 'speech_separation', or 'target_speaker_extraction')
        model_names: List of model names to use
        
    Returns:
        Dictionary with benchmark results for each chunk size
    """
    # Load audio file
    audio, sample_rate = librosa.load(audio_file, sr=None)
    
    # Create temporary directory for chunk files
    chunk_dir = os.path.join(temp_dir, 'chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    
    results = {}
    
    for chunk_size_ms in chunk_sizes:
        chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
        
        if chunk_size_samples >= len(audio):
            st.warning(f"Chunk size {chunk_size_ms}ms too large for audio file, skipping")
            continue
        
        # Extract 5 random chunks (or fewer if audio is short)
        max_chunks = 5
        max_start_pos = len(audio) - chunk_size_samples
        
        if max_start_pos <= 0:
            st.warning(f"Audio too short for chunk size {chunk_size_ms}ms, skipping")
            continue
            
        # Determine number of chunks to extract
        num_chunks = min(max_chunks, max(1, max_start_pos // chunk_size_samples))
        
        # Generate random starting points
        start_positions = sorted(random.sample(range(max_start_pos), num_chunks))
        
        chunk_results = []
        model_times = defaultdict(list)
        
        for i, start_pos in enumerate(start_positions):
            # Extract chunk
            audio_chunk = audio[start_pos:start_pos + chunk_size_samples]
            
            # Save chunk to file
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_size_ms}ms_{i}.wav")
            sf.write(chunk_path, audio_chunk, sample_rate)
            
            try:
                # Initialize ClearVoice with timing
                myClearVoice = ClearVoice(task=task, model_names=model_names)
                
                # Process the chunk and measure time
                start_time = time.time()
                output_wav = myClearVoice(input_path=chunk_path, online_write=False)
                total_time = time.time() - start_time
                
                # Get model-specific timing if available
                model_timings = getattr(myClearVoice, 'model_timings', {})
                
                # Store timings
                for model, model_time in model_timings.items():
                    model_times[model].append(model_time)
                
                # Add to chunk results
                chunk_result = {
                    'chunk_index': i,
                    'chunk_position_seconds': round(start_pos / sample_rate, 2),
                    'total_time': round(total_time, 3)
                }
                
                # Add model-specific times
                for model, timing in model_timings.items():
                    chunk_result[model] = timing
                
                chunk_results.append(chunk_result)
                
            except Exception as e:
                st.error(f"Error processing chunk {i} of size {chunk_size_ms}ms: {str(e)}")
        
        # Calculate averages for the chunk size
        if chunk_results:
            size_results = {}
            
            # Add average total time
            total_times = [result['total_time'] for result in chunk_results]
            size_results['total'] = {
                'avg_time': round(sum(total_times) / len(total_times), 3),
                'min_time': round(min(total_times), 3),
                'max_time': round(max(total_times), 3)
            }
            
            # Add model-specific averages
            for model, times in model_times.items():
                if times:
                    size_results[model] = {
                        'avg_time': round(sum(times) / len(times), 3),
                        'min_time': round(min(times), 3),
                        'max_time': round(max(times), 3)
                    }
            
            # Add individual chunk data
            size_results['chunks'] = chunk_results
            
            # Add to results
            results[str(chunk_size_ms)] = size_results
    
    # Clean up temporary chunk files
    try:
        for file in os.listdir(chunk_dir):
            os.remove(os.path.join(chunk_dir, file))
        os.rmdir(chunk_dir)
    except Exception as e:
        st.error(f"Error cleaning up chunk files: {str(e)}")
    
    return results

def display_benchmark_results(benchmark_results):
    """Display benchmark results in a nice format"""
    if not benchmark_results:
        st.warning("No benchmark results to display")
        return
    
    # Create tabs for different views
    benchmark_tabs = st.tabs(["Summary", "Detailed Results", "Charts"])
    
    with benchmark_tabs[0]:
        st.subheader("Benchmark Summary")
        
        # Create a summary dataframe
        summary_data = []
        
        for chunk_size, results in benchmark_results.items():
            row = {'Chunk Size (ms)': chunk_size}
            
            # Add total times
            if 'total' in results:
                row['Total Avg (ms)'] = results['total']['avg_time'] * 1000
                
            # Add model-specific times
            for model, times in results.items():
                if model not in ['total', 'chunks']:
                    row[f"{model} Avg (ms)"] = times['avg_time'] * 1000
            
            summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
        else:
            st.warning("No summary data available")
    
    with benchmark_tabs[1]:
        st.subheader("Detailed Results")
        
        # Create expandable sections for each chunk size
        for chunk_size, results in benchmark_results.items():
            with st.expander(f"Chunk Size: {chunk_size}ms"):
                # Show average times
                st.write("### Average Times")
                
                avg_data = []
                
                if 'total' in results:
                    avg_data.append({
                        'Component': 'Total',
                        'Avg Time (ms)': results['total']['avg_time'] * 1000,
                        'Min Time (ms)': results['total']['min_time'] * 1000,
                        'Max Time (ms)': results['total']['max_time'] * 1000
                    })
                
                for model, times in results.items():
                    if model not in ['total', 'chunks']:
                        avg_data.append({
                            'Component': model,
                            'Avg Time (ms)': times['avg_time'] * 1000,
                            'Min Time (ms)': times['min_time'] * 1000,
                            'Max Time (ms)': times['max_time'] * 1000
                        })
                
                if avg_data:
                    avg_df = pd.DataFrame(avg_data)
                    st.dataframe(avg_df)
                
                # Show individual chunk data
                if 'chunks' in results:
                    st.write("### Individual Chunks")
                    
                    chunk_data = []
                    
                    for chunk in results['chunks']:
                        row = {
                            'Chunk Index': chunk['chunk_index'],
                            'Position (s)': chunk['chunk_position_seconds'],
                            'Total Time (ms)': chunk['total_time'] * 1000
                        }
                        
                        # Add model-specific times
                        for key, value in chunk.items():
                            if key not in ['chunk_index', 'chunk_position_seconds', 'total_time']:
                                row[f"{key} (ms)"] = value * 1000
                        
                        chunk_data.append(row)
                    
                    if chunk_data:
                        chunk_df = pd.DataFrame(chunk_data)
                        st.dataframe(chunk_df)
    
    with benchmark_tabs[2]:
        st.subheader("Benchmark Charts")
        
        # Create chart comparing chunk sizes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for chart
        chunk_sizes = list(benchmark_results.keys())
        total_times = []
        
        for chunk_size in chunk_sizes:
            if 'total' in benchmark_results[chunk_size]:
                total_times.append(benchmark_results[chunk_size]['total']['avg_time'] * 1000)
            else:
                total_times.append(0)
        
        ax.bar(chunk_sizes, total_times)
        ax.set_xlabel('Chunk Size (ms)')
        ax.set_ylabel('Average Processing Time (ms)')
        ax.set_title('Processing Time by Chunk Size')
        
        st.pyplot(fig)
        
        # Model comparison chart for the largest chunk size
        if chunk_sizes:
            largest_chunk = max(chunk_sizes, key=int)
            results = benchmark_results[largest_chunk]
            
            model_names = []
            model_times = []
            
            for model, times in results.items():
                if model not in ['total', 'chunks']:
                    model_names.append(model)
                    model_times.append(times['avg_time'] * 1000)
            
            if model_names:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.bar(model_names, model_times)
                ax2.set_xlabel('Model')
                ax2.set_ylabel('Average Processing Time (ms)')
                ax2.set_title(f'Model Processing Times for {largest_chunk}ms Chunks')
                plt.xticks(rotation=45)
                
                st.pyplot(fig2)

def main():
    st.title("ClearerVoice Studio")
    
    # Create tabs for different tasks
    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction", "Benchmark"])
    
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
    
    # New Benchmark tab
    with tabs[3]:
        st.header("Model Benchmarking")
        
        # Task selection
        benchmark_task = st.selectbox(
            "Select Task to Benchmark", 
            ["speech_enhancement", "speech_separation", "target_speaker_extraction"]
        )
        
        # Model selection based on task
        if benchmark_task == "speech_enhancement":
            model_options = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        elif benchmark_task == "speech_separation":
            model_options = ['MossFormer2_SS_16K']
        else:  # target_speaker_extraction
            model_options = ['AV_MossFormer2_TSE_16K']
        
        benchmark_model = st.selectbox("Select Model", model_options, key="benchmark_model_selection")
        
        # Chunk size options
        chunk_sizes_input = st.text_input(
            "Chunk Sizes (ms, comma-separated)", 
            "100, 200, 500, 1000"
        )
        
        # File upload for benchmark
        benchmark_file = st.file_uploader(
            "Upload Audio/Video File for Benchmarking", 
            type=['wav', 'mp4', 'avi'], 
            key='benchmark'
        )
        
        if st.button("Run Benchmark", key='run_benchmark'):
            if benchmark_file is None:
                st.error("Please upload a file for benchmarking")
            else:
                # Parse chunk sizes
                try:
                    chunk_sizes = [int(size.strip()) for size in chunk_sizes_input.split(',') if size.strip()]
                    if not chunk_sizes:
                        st.error("Please enter valid chunk sizes")
                        return
                except ValueError:
                    st.error("Invalid chunk sizes. Please enter comma-separated numbers.")
                    return
                
                # Save file
                file_path = save_uploaded_file(benchmark_file)
                
                # Check if we need to extract audio from video
                if file_path.endswith(('.mp4', '.avi')):
                    if benchmark_task != "target_speaker_extraction":
                        with st.spinner("Extracting audio from video..."):
                            import cv2
                            import subprocess
                            
                            # Extract audio
                            audio_path = file_path.replace('.mp4', '.wav').replace('.avi', '.wav')
                            cmd = f"ffmpeg -i {file_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                            subprocess.call(cmd, shell=True)
                            
                            file_path = audio_path
                
                # Run benchmark
                with st.spinner(f"Running benchmark with {len(chunk_sizes)} chunk sizes..."):
                    benchmark_results = run_benchmark(
                        file_path, 
                        chunk_sizes, 
                        benchmark_task, 
                        [benchmark_model]
                    )
                
                # Display results
                if benchmark_results:
                    st.success("Benchmark completed successfully!")
                    display_benchmark_results(benchmark_results)
                else:
                    st.error("Benchmark produced no results. Check if the file and chunk sizes are compatible.")

if __name__ == "__main__":    
    main()