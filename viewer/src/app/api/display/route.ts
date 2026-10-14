import { displayResponse } from '../../../lib/results';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(request: Request) {
  return displayResponse(request);
}
