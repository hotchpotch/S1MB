import type { Meta, StoryObj } from '@storybook/nextjs-vite';
import { LeaderboardLoading } from './LeaderboardLoading';

const meta = {
  title: 'S1MB/Leaderboard loading',
  component: LeaderboardLoading,
  parameters: { layout: 'fullscreen' },
} satisfies Meta<typeof LeaderboardLoading>;
export default meta;
type Story = StoryObj<typeof meta>;
// Synthetic loading state; no measurements or progress estimates.
export const InitialLoad: Story = {};
