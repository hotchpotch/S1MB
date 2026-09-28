import type { NextConfig } from 'next';
const config: NextConfig = {
  trailingSlash: true,
  async rewrites() {
    return [{ source: '/storybook/', destination: '/storybook/index.html' }];
  },
};
export default config;
