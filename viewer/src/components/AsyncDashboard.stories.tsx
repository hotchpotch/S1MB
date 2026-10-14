import { useMemo } from 'react';
import type { Meta, StoryObj } from '@storybook/nextjs-vite';
import { AsyncDashboard } from './AsyncDashboard';
import dashboardMeta from './Dashboard.stories';

const snapshot = dashboardMeta.args.snapshot;
const meta = {
  title: 'S1MB/Async dashboard',
  component: AsyncDashboard,
  parameters: { layout: 'fullscreen', docs: { description: { component: 'Synthetic asynchronous loading states. No measured results or speed estimates.' } } },
} satisfies Meta<typeof AsyncDashboard>;
export default meta;
type Story = StoryObj<typeof meta>;
export const Loading: Story = { args: { load: () => new Promise(() => {}) } };
export const Failed: Story = { args: { load: () => Promise.reject(new Error('Synthetic request failure')) } };
export const Loaded: Story = { args: { load: () => Promise.resolve(snapshot) } };
export const Retry: Story = {
  render: function RetryStory() {
    const load = useMemo(() => {
      let attempt = 0;
      return () => ++attempt === 1 ? Promise.reject(new Error('Synthetic request failure')) : Promise.resolve(snapshot);
    }, []);
    return <AsyncDashboard load={load} />;
  },
  play: async ({ canvasElement }) => {
    const waitFor = async (condition: () => boolean) => {
      for (let i = 0; i < 100; i++) {
        if (condition()) return;
        await new Promise(resolve => setTimeout(resolve, 20));
      }
      throw new Error('Synthetic retry state did not settle.');
    };
    const retry = () => [...canvasElement.querySelectorAll('button')].find(button => button.textContent === 'Retry');
    await waitFor(() => !!retry());
    retry()!.click();
    await waitFor(() => !!canvasElement.querySelector('h2')?.textContent?.includes('Overall leaderboard'));
  },
};
