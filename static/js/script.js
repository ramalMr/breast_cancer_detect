// Fayl seçimi dəyişdikdə avtomatik olaraq formu göndər
document.getElementById('image').onchange = function() {
    document.getElementById('upload-form').submit();
};