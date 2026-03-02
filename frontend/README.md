# Motor Shield - Frontend

Web dashboard for the Motor Shield motor fault detection system. Built with Next.js 15, TypeScript, and HeroUI v2.

## Status

> **Note**: This frontend is currently a standalone UI application. It is **not yet connected** to the backend AI/ML service. Integration layer (REST API/WebSocket) is under development.

**Current Features:**
- Dashboard UI with motor health visualization
- Alert panel components
- Team and project information pages
- Responsive design with dark mode support
- Backend integration (coming soon)
- Real-time data visualization (coming soon)

## Technologies Used

- [Next.js 15](https://nextjs.org/) - React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) - Type-safe JavaScript
- [HeroUI v2](https://heroui.com/) - Modern React component library
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Framer Motion](https://www.framer.com/motion/) - Animation library
- [next-themes](https://github.com/pacocoursey/next-themes) - Theme management

## Prerequisites

- **Node.js**: 18.x or higher
- **npm**, **yarn**, **pnpm**, or **bun**

## Installation

### 1. Install Dependencies

```bash
# Using npm
npm install

# Using pnpm
pnpm install

# Using yarn
yarn install

# Using bun
bun install
```

### 2. Setup pnpm (Optional)

If using `pnpm`, ensure your `.npmrc` file contains:

```bash
public-hoist-pattern[]=*@heroui/*
```

Then reinstall dependencies:

```bash
pnpm install
```

## Development

### Run Development Server

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
# or
bun dev
```

The application will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## Project Structure

```
frontend/
├── app/                      # Next.js 15 App Router
│   ├── Dashboard/            # Motor health dashboard
│   ├── LearnMore/            # Information pages
│   ├── MeetTheTeam/          # Team profiles
│   ├── ProjectOverview/      # Project details
│   ├── components/           # Page-specific components
│   │   ├── AnimatedBackground.tsx
│   │   └── animations/
│   ├── layout.tsx            # Root layout
│   ├── page.tsx              # Home page
│   └── providers.tsx         # Theme provider
├── components/               # Shared components
│   ├── AlertsPanel.tsx
│   ├── GraphCard.tsx
│   ├── MotorHealthBar.tsx
│   ├── MotorStatus.tsx
│   ├── StatusSection.tsx
│   ├── navbar.tsx
│   └── theme-switch.tsx
├── config/                   # Configuration
│   ├── fonts.ts
│   └── site.ts
├── public/                   # Static assets
├── styles/                   # Global styles
│   └── globals.css
└── types/                    # TypeScript types
    └── index.ts
```

## Available Pages

- `/` - Home page
- `/Dashboard` - Motor health monitoring dashboard (UI only)
- `/MeetTheTeam` - Team member profiles
- `/LearnMore` - Project information
- `/ProjectOverview` - Technical overview

## Next Steps

To connect this frontend with the AI backend:

1. **Backend API Server**: Set up FastAPI or Flask server to expose ML inference endpoints
2. **API Client**: Create API client utilities in `frontend/lib/api`
3. **WebSocket Integration**: Implement real-time data streaming
4. **State Management**: Add data fetching and caching (React Query/SWR)
5. **Environment Variables**: Configure API endpoint URLs

Example setup:
```typescript
// lib/api/client.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function getMotorHealth(motorId: string) {
  const response = await fetch(`${API_BASE_URL}/motor/${motorId}/health`);
  return response.json();
}
```

## License

Part of the Motor Shield project. See the [root LICENSE](../LICENSE) file for details.
