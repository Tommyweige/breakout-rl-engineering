import './styles.css';

import { App } from './app/App';
import { initThemeToggle } from './theme';

const root = document.querySelector<HTMLElement>('#app');
if (!root) {
  throw new Error('missing #app root');
}

new App(root).mount();
initThemeToggle();
