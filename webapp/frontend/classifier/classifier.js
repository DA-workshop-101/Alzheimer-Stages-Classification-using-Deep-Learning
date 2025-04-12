const uploadInput = document.getElementById("upload");
const uploadBox = document.getElementById("uploadBox");
const previewCard = document.getElementById("previewCard");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const removeBtn = document.getElementById("removeFile");

// Handle image preview
uploadInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    fileName.textContent = file.name;
    previewImage.src = URL.createObjectURL(file);

    uploadBox.style.display = "none";
    previewCard.style.display = "block";
  }
});

removeBtn.addEventListener("click", () => {
  uploadInput.value = "";
  previewCard.style.display = "none";
  uploadBox.style.display = "flex";
});
