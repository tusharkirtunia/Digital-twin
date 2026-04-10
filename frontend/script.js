document.addEventListener('DOMContentLoaded', () => {

    // Toggle Login / Signup forms on Motivation Page
    const loginToggle = document.getElementById('loginToggle');
    if (loginToggle) {
        loginToggle.addEventListener('click', (e) => {
            e.preventDefault();
            const loginForm = document.getElementById('loginForm');
            const signupForm = document.getElementById('signupForm');

            if (loginForm.classList.contains('hidden')) {
                loginForm.classList.remove('hidden');
                signupForm.classList.add('hidden');
                loginToggle.textContent = 'Sign Up';
                document.querySelector('.member-text').childNodes[0].nodeValue = 'New to VAYURA? ';
            } else {
                loginForm.classList.add('hidden');
                signupForm.classList.remove('hidden');
                loginToggle.textContent = 'Log In';
                document.querySelector('.member-text').childNodes[0].nodeValue = 'Already a Member? ';
            }
        });
    }

    // Handle Login Submit -> Redirect to Streamlit Dashboard at localhost:8501
    const submitLoginBtn = document.getElementById('submitLoginBtn');
    if (submitLoginBtn) {
        submitLoginBtn.addEventListener('click', (e) => {
            e.preventDefault();

            const emailInput = document.querySelector('#loginForm input[type="email"]');
            const passInput = document.querySelector('#loginForm input[type="password"]');

            if (!emailInput.value || !passInput.value) {
                alert("Please enter both email and password.");
                return;
            }

            submitLoginBtn.textContent = 'Authenticating...';
            submitLoginBtn.style.opacity = '0.7';
            submitLoginBtn.style.cursor = 'wait';

            setTimeout(() => {
                // Generate a mock auth token
                const token = 'vayura-auth-token-' + Math.random().toString(36).substr(2, 9);
                const userExtracted = encodeURIComponent(emailInput.value);
                // Redirect to HealthTwin platform with token in URL
                window.location.href = `http://localhost:8501/?token=${token}&user=${userExtracted}`;
            }, 800);
        });
    }

    // Set Navbar Log In button to also switch to login form if on index.html
    const navLoginBtns = document.querySelectorAll('.btn-outline');
    navLoginBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (window.location.pathname.includes('index.html') || window.location.pathname === '/') {
                const loginForm = document.getElementById('loginForm');
                const signupForm = document.getElementById('signupForm');
                if (loginForm && loginForm.classList.contains('hidden')) {
                    loginForm.classList.remove('hidden');
                    signupForm.classList.add('hidden');
                    if (loginToggle) {
                        loginToggle.textContent = 'Sign Up';
                        document.querySelector('.member-text').childNodes[0].nodeValue = 'New to VAYURA? ';
                    }
                }
            } else {
                // If on another page, go to index.html with hash to open login
                window.location.href = 'index.html#login';
            }
        });
    });

    // Check for #login hash on load
    if (window.location.hash === '#login' && loginToggle) {
        // trigger click to open
        loginToggle.click();
    }

    // Handle Contact Form Submit
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const btn = contactForm.querySelector('.btn-black');
            const originalText = btn.textContent;

            btn.textContent = 'Sending...';
            btn.style.opacity = '0.7';

            setTimeout(() => {
                btn.textContent = 'Sent Successfully!';
                btn.style.backgroundColor = '#1dd1a1';
                contactForm.reset();

                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.backgroundColor = '#000';
                    btn.style.opacity = '1';
                }, 3000);
            }, 1000);
        });
    }

    // Add subtle entrance animations to elements
    const fadeElements = document.querySelectorAll('.auth-box, .contact-card, .devices-mockup');
    fadeElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';

        setTimeout(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 100 + (index * 150));
    });

    // Handle "Sign Up With Email" button to switch to email fields
    const signupEmailBtn = document.getElementById('signupEmailBtn');
    if (signupEmailBtn) {
        signupEmailBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const loginToggle = document.getElementById('loginToggle');
            if (loginToggle) {
                loginToggle.click(); // Switches view to email inputs

                // Change button text dynamically to indicate Sign Up flow
                const submitBtn = document.getElementById('submitLoginBtn');
                if (submitBtn) {
                    submitBtn.textContent = 'Create Account & Access Platform';
                }
            }
        });
    }

    // Reset submit button text if they manual toggle back and forth
    const loginToggle2 = document.getElementById('loginToggle');
    if (loginToggle2) {
        loginToggle2.addEventListener('click', () => {
            const submitBtn = document.getElementById('submitLoginBtn');
            const loginForm = document.getElementById('loginForm');
            if (submitBtn && loginForm) {
                if (!loginForm.classList.contains('hidden') && loginToggle2.textContent === 'Log In') {
                    // It means they toggled to Sign Up logic via the link
                    submitBtn.textContent = 'Create Account & Access Platform';
                } else if (!loginForm.classList.contains('hidden') && loginToggle2.textContent === 'Sign Up') {
                    submitBtn.textContent = 'Log In to Platform';
                }
            }
        });
    }
});
