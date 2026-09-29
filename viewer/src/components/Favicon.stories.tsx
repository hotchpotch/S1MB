import type { Meta, StoryObj } from '@storybook/nextjs-vite';
import favicon from '../../public/favicon.svg';

const meta = {
  title: 'S1MB/Favicon',
  parameters: { layout: 'centered' },
  render: () => <div className="flex flex-wrap gap-6">
    {['#f8faf9', '#101b16'].map(background => <div key={background} style={{ background }} className="flex items-center gap-6 rounded-lg p-6">
      {[16, 32, 64, 128].map(size => <img key={size} src={favicon.src} width={size} height={size} alt={`S1MB favicon at ${size}px`} />)}
    </div>)}
  </div>,
} satisfies Meta;
export default meta;
export const SizesAndBackgrounds: StoryObj<typeof meta> = {};
