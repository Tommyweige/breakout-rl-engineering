import './styles.css';

import { App } from './app/App';

const root = document.querySelector<HTMLElement>('#app');
if (!root) {
  throw new Error('missing #app root');
}

new App(root).mount();
