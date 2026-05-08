import { ConsoleClient } from "../components/ConsoleClient";
import { getDashboardRuns } from "../lib/data";

export default async function Page({
  searchParams
}: {
  searchParams: Promise<{ fixture?: string }>;
}) {
  const params = await searchParams;
  const runs = await getDashboardRuns(params.fixture === "1");
  return <ConsoleClient run={runs[0]} runs={runs} />;
}
