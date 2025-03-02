class ApiClient {
    constructor(baseUrl = 'https://legaldocs-api.onrender.com') {  // Update with your Render URL
        this.baseUrl = baseUrl;
    }

    async processDocument(file, options = { anonymize: true, summarize: true }) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('anonymize', options.anonymize);
        formData.append('summarize', options.summarize);

        const response = await fetch(`${this.baseUrl}/api/process`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText} (${response.status})`);
        }

        return await response.json();
    }

    async processSampleDocument(sampleType, options = { anonymize: true, summarize: true }) {
        const response = await fetch(`${this.baseUrl}/api/process-sample`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sampleType, anonymize: options.anonymize, summarize: options.summarize })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.statusText} (${response.status})`);
        }

        return await response.json();
    }
}

const api = new ApiClient();