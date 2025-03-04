document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent form from reloading the page

    let fileInput = document.getElementById("imageInput");
    let previewImage = document.getElementById("previewImage");
    let resultDiv = document.getElementById("result");

    // ✅ Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    // ✅ Show image preview before sending request
    let reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewImage.style.display = "block"; // Ensure image is visible
    };
    reader.readAsDataURL(fileInput.files[0]);

    try {
        // ✅ Send the image to Flask API for prediction
        let response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        let result = await response.json();
        console.log("🔹 API Response:", result); // Debugging log

        // ✅ Handle API response
        if (result.error) {
            resultDiv.innerText = "❌ Error: " + result.error;
            resultDiv.style.color = "red";
        } else {
            resultDiv.innerHTML = `
                <strong>Predicted Disease:</strong> ${result.predicted_disease} <br> 
                <strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%
            `;
            resultDiv.style.color = "green";
        }
    } catch (error) {
        console.error("🔴 Fetch Error:", error);
        resultDiv.innerText = "❌ Error: Unable to get prediction.";
        resultDiv.style.color = "red";
    }
});
