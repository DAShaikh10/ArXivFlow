// Loading / error / empty placeholders.

export function Loading({ message = "Loading…" }: { message?: string }) {
  return (
    <div className="af-state">
      <div className="af-spinner" />
      {message}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="af-state">
      <span className="af-error">{message}</span>
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <div className="af-state">{message}</div>;
}

export function FeedSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="feed-list">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="af-skel" />
      ))}
    </div>
  );
}
