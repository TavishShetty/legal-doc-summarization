document.addEventListener('DOMContentLoaded', () => {
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    const resultsContainer = document.getElementById('results-container');

    // Upload Form
    document.getElementById('document-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        loadingModal.show();

        const file = document.getElementById('document-upload').files[0];
        const options = {
            anonymize: document.getElementById('anonymize-check').checked,
            summarize: document.getElementById('summarize-check').checked
        };

        try {
            const result = await api.processDocument(file, options);
            displayResults(result);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loadingModal.hide();
        }
    });

    // Sample Form
    document.getElementById('process-sample-btn').addEventListener('click', async () => {
        const sampleType = document.getElementById('sample-select').value;
        if (!sampleType) {
            alert('Please select a sample document.');
            return;
        }

        loadingModal.show();
        const options = {
            anonymize: document.getElementById('sample-anonymize-check').checked,
            summarize: document.getElementById('sample-summarize-check').checked
        };

        try {
            const result = await api.processSampleDocument(sampleType, options);
            displayResults(result);
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loadingModal.hide();
        }
    });

    function displayResults(result) {
        document.getElementById('original-content').textContent = result.original || 'N/A';
        document.getElementById('anonymized-content').textContent = result.anonymized || 'Not processed';
        document.getElementById('summary-content').textContent = result.summary || 'Not processed';
        resultsContainer.classList.remove('d-none');
    }
});