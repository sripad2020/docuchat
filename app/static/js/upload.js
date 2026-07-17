document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const statusText = document.getElementById('statusText');
    const statusDetail = document.getElementById('statusDetail');
    const actionContainer = document.getElementById('actionContainer');
    const chatLink = document.getElementById('chatLink');

    if (!dropzone) return;

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add('border-brand-purple', 'bg-brand-purple-light');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove('border-brand-purple', 'bg-brand-purple-light');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        handleFiles(files);
    });

    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (file.type !== 'application/pdf') {
            alert('Please upload a valid PDF file.');
            return;
        }

        uploadFile(file);
    }

    async function uploadFile(file) {
        dropzone.classList.add('hidden');
        progressContainer.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/v1/documents', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Upload failed');
            }
            
            const data = await res.json();
            const docId = data.id;
            
            pollStatus(docId);
            
        } catch (error) {
            statusText.innerText = 'Error';
            statusDetail.innerText = error.message;
            statusText.classList.add('text-red-600');
            progressBar.classList.add('bg-red-500');
        }
    }

    function pollStatus(docId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/documents/${docId}/status`);
                if (!res.ok) throw new Error('Failed to fetch status');
                
                const data = await res.json();
                const status = data.status;
                
                statusText.innerText = `Status: ${status.charAt(0).toUpperCase() + status.slice(1)}`;
                
                let progress = 0;
                if (status === 'queued') progress = 10;
                else if (status === 'extracting') progress = 30;
                else if (status === 'embedding') progress = 60;
                else if (status === 'indexing') progress = 85;
                else if (status === 'ready') progress = 100;
                else if (status === 'failed') {
                    progress = 100;
                    progressBar.classList.add('bg-red-500');
                    statusText.classList.add('text-red-600');
                    statusDetail.innerText = data.error || 'Unknown error';
                    clearInterval(interval);
                    return;
                }
                
                progressBar.style.width = `${progress}%`;
                
                if (status === 'ready') {
                    clearInterval(interval);
                    progressBar.classList.add('bg-green-500');
                    progressBar.classList.remove('bg-brand-purple');
                    statusText.innerText = 'Ready!';
                    statusText.classList.add('text-green-600');
                    actionContainer.classList.remove('hidden');
                    chatLink.href = `/chat/${docId}`;
                }
                
            } catch(e) {
                console.error("Polling error", e);
            }
        }, 2000);
    }
});
