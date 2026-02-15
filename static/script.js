const links = document.querySelectorAll("[data-scroll]");

links.forEach((link) => {
	link.addEventListener("click", (event) => {
		const targetId = link.getAttribute("href");
		if (!targetId || !targetId.startsWith("#")) {
			return;
		}
		const target = document.querySelector(targetId);
		if (!target) {
			return;
		}
		event.preventDefault();
		target.scrollIntoView({ behavior: "smooth", block: "start" });
	});
});

const authAlert = document.querySelector("[data-auth-alert]");
const authForm = document.querySelector("[data-auth-form]");

const updateAuthAlert = (message, tone) => {
	if (!authAlert) {
		return;
	}
	authAlert.textContent = message;
	authAlert.classList.remove("is-error", "is-success");
	if (tone) {
		authAlert.classList.add(tone === "success" ? "is-success" : "is-error");
	}
	authAlert.hidden = false;
};

if (authAlert) {
	const params = new URLSearchParams(window.location.search);
	const errorMessage = params.get("error");
	const successMessage = params.get("success");

	if (errorMessage || successMessage) {
		updateAuthAlert(errorMessage || successMessage, errorMessage ? "error" : "success");
	}
}

if (authForm) {
	authForm.addEventListener("submit", async (event) => {
		event.preventDefault();
		const formData = new FormData(authForm);
		try {
			const response = await fetch(authForm.action, {
				method: "POST",
				body: formData,
				headers: {
					"X-Requested-With": "XMLHttpRequest",
				},
			});
			const data = await response.json();
			if (!response.ok || !data.success) {
				updateAuthAlert(data.message || "Something went wrong. Please try again.", "error");
				return;
			}
			if (data.redirect) {
				window.location.href = data.redirect;
				return;
			}
			updateAuthAlert(data.message || "Success.", "success");
		} catch (error) {
			updateAuthAlert("Network error. Please try again.", "error");
		}
	});
}
