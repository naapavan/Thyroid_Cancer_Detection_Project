
        function handleFileSelect(event) {
            event.preventDefault();
            var fileInput = document.getElementById('imageInput');
            var fileNameDisplay = document.getElementById('fileName');

            var files = event.target.files || event.dataTransfer.files;

            if (files.length > 0) {
                fileNameDisplay.textContent = files[0].name;
            } else {
                fileNameDisplay.textContent = 'No file chosen';
            }

            // Display summary only if predicted_class is defined
            if (document.getElementById('predicted_class').innerText.trim() !== '') {
                displaySummary();
            }
        }

        function browseFile() {
            document.getElementById('imageInput').click();
        }

        function validateForm() {
            var fileInput = document.getElementById('imageInput');
            var fileNameDisplay = document.getElementById('fileName');

            if (fileInput.files.length === 0) {
                fileNameDisplay.textContent = 'Choose the file first';
                return false; // Prevent form submission
            }

            // Continue with form submission
            return true;
        }

        function displaySummary() {
            var predictedClass = document.getElementById('predicted_class').innerText;
            var summary = "";

            // Add summaries based on predicted classes
            switch(predictedClass) {
                case "normal thyroid":
                    summary = "(Normal): This category indicates a normal thyroid nodule, showing no suspicious characteristics. It is considered non-cancerous, and no further action is typically required.";
                    break;
                case "Benign":
                    summary = "(Benign): Nodules in this category are classified as benign, meaning they are not likely to be cancerous. This category often includes nodules with typical features of a benign nature. Follow-up monitoring may be recommended to ensure stability.This category is used when nodules have features that make them likely to be benign but with a small possibility of malignancy. Follow-up monitoring may be recommended to ensure stability.";
                    
                    break;
                
                case "4A":
                    summary = "(TIRADS-4A): This category suggests that there are weak or minimal signs that the nodule may be malignant. Further diagnostic evaluation, such as a fine-needle aspiration (FNA) biopsy, may be recommended to better assess the risk.";
                    break;
                case "4B":
                    summary = "(TIRADS-4B): Nodules in this category show more distinct signs that raise concerns about malignancy. Additional diagnostic procedures, such as biopsy, are often recommended for further evaluation.";
                    break;
                case "4C":
                    summary = "(TIRADS-4C): This category indicates a high likelihood of malignancy. Urgent diagnostic measures, such as biopsy or surgical evaluation, are typically recommended for nodules falling into this category.";
                    break;
                case "5":
                    summary = "(TIRADS-5): Nodules classified as TIRADS-5 are highly suspicious for malignancy based on imaging features. Immediate further investigation and management, such as biopsy or surgery, are usually recommended.";
                    break;
                // Add more cases for other predicted classes
            }

            // Display the summary
            document.getElementById('predictionSummary').innerText = summary;
        }