const actionBtn = document.getElementById("actionBtn");

if (actionBtn) {
	actionBtn.addEventListener("click", () => {
		actionBtn.textContent = "You clicked it";
	});
}
