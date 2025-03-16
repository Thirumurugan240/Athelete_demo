document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.querySelector('input[type="file"]');
    const submitButton = document.querySelector('button');

    fileInput.addEventListener("change", function() {
        if (fileInput.files.length > 0) {
            submitButton.style.background = "#00ff99"; 
        }
    });

    submitButton.addEventListener("mouseenter", function() {
        submitButton.style.transform = "scale(1.1)";
    });

    submitButton.addEventListener("mouseleave", function() {
        submitButton.style.transform = "scale(1)";
    });
});
