// Logo swap — call when ready
const logoBox = document.querySelector('.logo-box');

function setLogo(src, alt = 'Logo') {
  logoBox.innerHTML = `<img src="${src}" alt="${alt}" style="width:100%;height:100%;object-fit:contain;border-radius:8px;" />`;
}

// setLogo('images/logo.png', 'My Brand');